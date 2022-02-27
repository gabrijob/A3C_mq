#/!/bin/bash

#mq="message_queue_1"

static1(){

	#qosmax=4
	loss=5 #no use 
        qosmin=1 #no use
        rep=1
		./prepare_env.sh 4 1 5 $interface $infiniband
#                name="BP_8-8-9_static-$qosmax-eno"
                sockets=4
                #  1 to ib0, "" to eno
                infiniband=""
                #interfaces at dahu cluser are ib0 or enp24s0f0
                n=1;
		#for app in "SUMServer" "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
                for app in "SUMServer"; do
                   for i in 45; do
		      #for i in 5 17 45; do
			    qosmax=$i
			    name="BP_8-8-9_saqn-$mq-eno"
                        while [ $n -le $rep ]; do
                                ./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq

                                sleep 1
                                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                                sleep 1
                                ./logs.sh
                                sleep 1
                                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                   		mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
		   		n=$((n+1))
                                sleep 1
                        done
                        n=1
                    done
                done
	}
static2(){	
		loss=5 #no use 
        qosmin=1 #no use
        rep=1

		./prepare_env.sh 8 4 9 $interface $infiniband
                name="BP_8-4-9_static-eno"
                sockets=2
                #  1 to ib0, "" to eno
                infiniband=""
                #interfaces at dahu cluser are ib0 or enp24s0f0
                n=1;
		 for app in "StatelessSUMServer"; do
			for i in 17 45; do
			    qosmax=$i
                        while [ $n -le $rep ]; do
                                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq

                                sleep 1
                                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                                sleep 1
                                ./logs.sh
                                sleep 1
                                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                n=$((n+1))
                                sleep 1
                        done
                        n=1
                    done
                done
	}
static(){
  loss=5 #no use 
        qosmin=1 #no use
        rep=1

                ./prepare_env.sh 8 1 9 $interface $infiniband
                name="BP_8-1-9_static-eno1"
                sockets=8
                #  1 to ib0, "" to eno
                infiniband=""
                #interfaces at dahu cluser are ib0 or enp24s0f0
                n=1 
                for app in "GlobalHistogramServer" "StatelessSUMServer"; do
	          for i in 17 45 5; do
                  	qosmax=$i
		    	  while [ $n -le $rep ]; do
                                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq

                                sleep 1
                                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                                sleep 1
                                ./logs.sh
                                sleep 1
                                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                n=$((n+1))
                                sleep 1
                        done
                        n=1
                    done
                done

}


mq="message_queue_dynamic"
#mq="message_queue"
s(){
	qosmax=4
	loss=5
	qosmin=1
	rep=1
		./prepare_env.sh 8 4 9 $interface $infiniband
		name="BP_8-4-9_max-$qosmax-min-$qosmin-$loss-eno"
		sockets=2
		#  1 to ib0, "" to eno
		infiniband=""
		#interfaces at dahu cluser are ib0 or enp24s0f0
		n=1; 
		for app in  "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
 		    for i in $qosmin; do
        		while [ $n -le $rep ]; do
				./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq

				sleep 1
		                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                		sleep 1
		                ./logs.sh
                		sleep 1
		                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                		mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
		                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                		n=$((n+1))
		                sleep 1
		        done
        		n=1
    		    done
		done

  ./prepare_env.sh 8 1 9 $interface $infiniband
                name="BP_8-1-9_max-$qosmax-min-$qosmin-$loss-eno"
                sockets=8
                #  1 to ib0, "" to eno
                infiniband=""
                #interfaces at dahu cluser are ib0 or enp24s0f0
                n=1; 
                for app in  "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
                    for i in $qosmin; do
                        while [ $n -le $rep ]; do
                                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq

                                sleep 1
                                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                                sleep 1
                                ./logs.sh
                                sleep 1
                                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                                n=$((n+1))
                                sleep 1
                        done
                        n=1
                    done
                done

}







sa(){

qosmax=8
loss=5
qosmin=4
rep=1



name="BP_8-4-9_max-$qosmax-min-$qosmin-$loss-eno"

sockets=2


#  1 to ib0, "" to eno
infiniband=""
interface="ib0"
#"enp24s0f0"
./prepare_env.sh 8 4 9 $interface $infiniband

n=1; 

for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin; do 
	while [ $n -le $rep ]; do 
		./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
		sleep 1
		mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n; 
		sleep 1
		./logs.sh
		sleep 1
		mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
		mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
		n=$((n+1));
		sleep 1 
	done; 
	n=1; 
    done
done

name="BP_8-1-9_max-$qosmax-min-$qosmin-$loss-eno"

sockets=8


./prepare_env.sh 8 1 9 $interface $infiniband

n=1;

for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done


}

sa1(){

#qosmax=10
loss=5
qosmin=8
rep=1



./prepare_env.sh 8 4 9 $interface $infiniband

name="BP_8-4-9_max-$qosmax-min-$qosmin-$loss-eno"

sockets=2

#  1 to ib0, "" to eno
infiniband=""
interface="ib0"
#"enp24s0f0"

n=1;

for app in "SUMServer" "StatelessSUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in 12 16 20; do
	    qosmax=$i
        while [ $n -le $rep ]; do
                ./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

./prepare_env.sh 8 1 9 $interface $infiniband

name="BP_8-1-9_max-$qosmax-min-$qosmin-$loss-eno"

sockets=8

#  1 to ib0, "" to eno
infiniband=""
interface="ib0"
#"enp24s0f0"

n=1;

for app in "SUMServer" "StatelessSUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in 12 16 20; do
            qosmax=$i
        while [ $n -le $rep ]; do
                ./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done







}

sa2(){

#qosmax=10
loss=5
qosmin=8
rep=1
for pn in 8 4 1; do

./prepare_env.sh 8 $pn 9 $interface $infiniband

name="BP_8-$pn-9_max-$qosmaxmin_min-$qosmin-$loss-eno"

sockets=$soc_aux


#  1 to ib0, "" to eno
infiniband=1
interface="ib0"
#"enp24s0f0"

name="BP_889_min-$qosmin-$loss-ib0-"
n=1;

for app in "StatelessSUMServer" "GlobalSUMServer" "GlobalHistogramServer" "SUMServer"; do
    for i in 16 20 12; do
            qosmax=$i
        while [ $n -le $rep ]; do
                ./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done
sockets=$((soc_aux*4))
soc_aux=$((sock_aux+1))

done
}

sa3(){

qosmax=8
loss=5
qosmin=4
rep=1

for pn in 8 4 1; do

./prepare_env.sh 8 $pn 9 $interface $infiniband

name="BP_8-$pn-9_max-$qosmaxmini-min-$qosmin-$loss-eno"

sockets=$soc_aux



#  1 to ib0, "" to eno
infiniband=""
interface="ib0"
#"enp24s0f0"

n=1;

for app in "StatelessSUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done
sockets=$((soc_aux*4))
soc_aux=$((sock_aux+1))


done
}



s1(){

sockets=1

#  1 to ib0, "" to eno
infiniband=1
interface="ib0"

./prepare_env.sh 8 8 9 $interface $infiniband

name="BP_888_dynamic_ib0_8-50-block400-window-6000"
n=1;
#for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
for app in "SUMServer" "StatelessSUMServer"; do
	for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets $infiniband $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}
s2(){


#  1 to ib0, "" to eno
infiniband=""
interface="ib0"
for pn in 8 4 1; do

./prepare_env.sh 8 $pn 9 $interface $infiniband

name="BP_8-$pn-9_max-$qosmaxmin_min-$qosmin-$loss-eno"

sockets=$soc_aux


name="BP_848_dynamic_eno"
n=1;
for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
   for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq 
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

done
}

s3(){

sockets=2

#  1 to ib0, "" to eno
infiniband=1
interface="ib0"

./prepare_env.sh 8 4 9 $interface $infiniband

name="BP_848_dynamic_ib0"
n=1;
for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets $infiniband $qosmax $loss $mq
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}


s4(){

sockets=4

#  1 to ib0, "" to eno
infiniband=""
interface="ib0"

./prepare_env.sh 8 2 9 $interface $infiniband

name="BP_828_dynamic_eno"
n=1;
for app in "GlobalHistogramServer"; do
    for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}

s5(){

sockets=4

#  1 to ib0, "" to eno
infiniband=1
interface="ib0"

./prepare_env.sh 8 2 9 $interface $infiniband

name="BP_828_dynamic_ib0"
n=1;
for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin ; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets $infiniband $qosmax $loss
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}

s6(){

sockets=8

#  1 to ib0, "" to eno
infiniband=""
interface="ib0"

./prepare_env.sh 8 1 9 $interface $infiniband

name="BP_818_dynamic_eno"
n=1;
for app in "GlobalHistogramServer"; do
    for i in $qosmin; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}

s7(){

sockets=8

#  1 to ib0, "" to eno
infiniband=1
interface="ib0"

#./prepare_env.sh 8 1 9 $interface $infiniband

name="BP_818_synamic_ib0"
n=1;
for app in "StatelessSUMServer" "SUMServer" "GlobalSUMServer" "GlobalHistogramServer"; do
    for i in $qosmin ; do
        while [ $n -le $rep ]; do
                ./start_apps.sh $i $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets $infiniband $qosmax $loss
                sleep 1
                mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
                sleep 1
                ./logs.sh
                sleep 1
                mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
                mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
                n=$((n+1));
                sleep 1
        done;
        n=1;
    done
done

}


saqn1(){
	mq="message_queue_0"
	#qosmax=4
	loss=5 #no use 
	qosmin=1 #no use
	rep=1
	./prepare_env.sh 8 4 9 $interface $infiniband
	#                name="BP_8-8-9_static-$qosmax-eno"
	sockets=2
	#  1 to ib0, "" to eno
	infiniband=""
	#interfaces at dahu cluser are ib0 or enp24s0f0
	n=1;
	#for app in "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
	for app in "StatelessSUMServer"; do	
	#for app in "SUMServer"; do
		for i in 32; do
		#for i in 5 17 45; do
			qosmax=$i
			name="BP_8-4-9_saqn-$mq-eno-untrained-2h"
			while [ $n -le $rep ]; do
				./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
				sleep 1
				mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
				sleep 1
				./logs.sh
				sleep 1
				mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
				n=$((n+1))
				sleep 1
			done;
			n=1
		done
	done
}


saqn2(){
	mq="message_queue_1"
	#qosmax=4
	loss=5 #no use 
	qosmin=1 #no use
	rep=1
	./prepare_env.sh 8 1 9 $interface $infiniband
	#                name="BP_8-8-9_static-$qosmax-eno"
	sockets=8
	#  1 to ib0, "" to eno
	infiniband=""
	#interfaces at dahu cluser are ib0 or enp24s0f0
	n=1;
	for app in  "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
	#for app in "SUMServer"; do
		for i in 32; do
		#for i in 5 17 45; do
			qosmax=$i
			name="BP_8-8-9_saqn-$mq-eno-trained-2h"
			while [ $n -le $rep ]; do
				./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
				sleep 1
				mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
				sleep 1
				./logs.sh
				sleep 1
				mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
				n=$((n+1))
				sleep 1
			done;
			n=1
		done
	done
}


saqn3(){
	mq="message_queue_2"
	#qosmax=4
	loss=5 #no use 
	qosmin=1 #no use
	rep=1
	./prepare_env.sh 8 8 9 $interface $infiniband
	#                name="BP_8-8-9_static-$qosmax-eno"
	sockets=1
	#  1 to ib0, "" to eno
	infiniband=""
	#interfaces at dahu cluser are ib0 or enp24s0f0
	n=1;
	#for app in "SUMServer" "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
	for app in "SUMServer"; do
		for i in 32; do
		#for i in 5 17 45; do
			qosmax=$i
			name="BP_8-8-9_saqn-$mq-eno-untrained-2h"
			while [ $n -le $rep ]; do
				./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
				sleep 1
				mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
				sleep 1
				./logs.sh
				sleep 1
				mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
				n=$((n+1))
				sleep 1
			done;
			n=1
		done
	done
}


saqn4(){
	mq="message_queue_3"
	#qosmax=4
	loss=5 #no use 
	qosmin=1 #no use
	rep=1
	./prepare_env.sh 8 8 9 $interface $infiniband
	#                name="BP_8-8-9_static-$qosmax-eno"
	sockets=1
	#  1 to ib0, "" to eno
	infiniband=""
	#interfaces at dahu cluser are ib0 or enp24s0f0
	n=1;
	#for app in "SUMServer" "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
	for app in "SUMServer"; do
		for i in 32; do
		#for i in 5 17 45; do
			qosmax=$i
			name="BP_8-8-9_saqn-$mq-eno-untrained-2h"
			while [ $n -le $rep ]; do
				./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
				sleep 1
				mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
				sleep 1
				./logs.sh
				sleep 1
				mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
				n=$((n+1))
				sleep 1
			done;
			n=1
		done
	done
}

a3c1(){
	mq="message_queue_4"
	#qosmax=4
	loss=5 #no use 
	qosmin=1 #no use
	rep=1
	./prepare_env.sh 8 1 9 $interface $infiniband
	#                name="BP_8-8-9_static-$qosmax-eno"
	sockets=1
	#  1 to ib0, "" to eno
	infiniband=""
	#interfaces at dahu cluser are ib0 or enp24s0f0
	n=1;
	#for app in "SUMServer" "GlobalSUMServer" "GlobalHistogramServer" "StatelessSUMServer"; do
	for app in "SUMServer"; do
		for i in 32; do
		#for i in 5 17 45; do
			qosmax=$i
			name="BP_8-1-9_saqn-$mq-eno-A3C-2h"
			while [ $n -le $rep ]; do
				./start_apps.sh $qosmin $app '--conf "spark.driver.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/DRIVER.log" --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/tmp/EXECUTOR.log"' $sockets "" $qosmax $loss $mq
				sleep 1
				mkdir LOGS/$name'_'$app'_'$i'_'GB-rep$n;
				sleep 1
				./logs.sh
				sleep 1
				mv LOGS/*.csv LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.txt LOGS/$name'_'$app'_'$i'_'GB-rep$n
				mv LOGS/*.log LOGS/$name'_'$app'_'$i'_'GB-rep$n
				#mv LOGS/saqn_* LOGS/$name'_'$app'_'$i'_'GB-rep$n
				n=$((n+1))
				sleep 1
			done;
			n=1
		done
	done
}


a3c1
#saqn1
#saqn2
#saqn3
#saqn4
#s
#sa
#sa1
#sa2
#sa3



#s
#s2
#s4
#s6
#s1
#s3
#s5
#s7







